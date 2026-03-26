import argparse
import csv
import json
import math
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

VISUAL_VERTICAL_EXAGGERATION = 1.35
VISUAL_ZOOM = 1.18
CAMERA_ELEV = 8.0
VIEW_PERP_DOT_TOL = 2.0e-2
ARROW_CLEARANCE_RATIO = 0.08
ARROW_CLEARANCE_MIN = 0.40
ARROW_CLEARANCE_MAX_FACTOR = 4.0
ARROW_COUNT = 3
LEFT_TO_RIGHT_REQUIRED = True


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _display_coords(x_value, y_value, z_value):
    return [float(x_value), float(z_value), float(y_value) * VISUAL_VERTICAL_EXAGGERATION]


def _display_vector(vec3):
    return np.asarray(
        [float(vec3[0]), float(vec3[2]), float(vec3[1]) * VISUAL_VERTICAL_EXAGGERATION],
        dtype=float,
    )


def _unit_vector(vec):
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-12:
        return np.asarray([0.0, 0.0, 0.0], dtype=float)
    return np.asarray(vec, dtype=float) / norm


def _wind_vector_from_axis(axis_text: str):
    axis_key = str(axis_text or "+X").strip().upper()
    mapping = {
        "+X": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "-X": np.asarray([-1.0, 0.0, 0.0], dtype=float),
        "+Y": np.asarray([0.0, 1.0, 0.0], dtype=float),
        "-Y": np.asarray([0.0, -1.0, 0.0], dtype=float),
        "+Z": np.asarray([0.0, 0.0, 1.0], dtype=float),
        "-Z": np.asarray([0.0, 0.0, -1.0], dtype=float),
    }
    return mapping.get(axis_key, np.asarray([1.0, 0.0, 0.0], dtype=float))


def _build_face_polygons(mesh, node_map):
    polygons = []
    element_labels = []
    for element in mesh["elements"]:
        coords = []
        for node_id in element["connectivity"]:
            node = node_map[node_id]
            coords.append(_display_coords(node["x"], node["y"], node["z"]))
        polygons.append(coords)
        element_labels.append(element["label"])
    return polygons, element_labels


def _set_view_box(ax, points, zoom=VISUAL_ZOOM):
    arr = np.asarray(points, dtype=float)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0e-9)
    centers = 0.5 * (mins + maxs)
    half = 0.5 * spans / max(float(zoom), 1.0e-6)
    pad = 0.05 * spans
    ax.set_xlim(centers[0] - half[0] - pad[0], centers[0] + half[0] + pad[0])
    ax.set_ylim(centers[1] - half[1] - pad[1], centers[1] + half[1] + pad[1])
    ax.set_zlim(centers[2] - half[2] - pad[2], centers[2] + half[2] + pad[2])
    try:
        ax.set_box_aspect((spans[0], spans[1], spans[2]))
    except Exception:
        pass


def _view_vectors_from_angles(elev_deg: float, azim_deg: float):
    elev_rad = math.radians(float(elev_deg))
    azim_rad = math.radians(float(azim_deg))

    camera_from_target = np.asarray(
        [
            math.cos(elev_rad) * math.cos(azim_rad),
            math.cos(elev_rad) * math.sin(azim_rad),
            math.sin(elev_rad),
        ],
        dtype=float,
    )
    forward = _unit_vector(-camera_from_target)  # camera -> target

    world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    right = _unit_vector(np.cross(forward, world_up))
    if float(np.linalg.norm(right)) < 1.0e-9:
        right = _unit_vector(np.cross(forward, np.asarray([0.0, 1.0, 0.0], dtype=float)))
    up = _unit_vector(np.cross(right, forward))
    return forward, right, up


def _camera_policy_from_wind(wind_display_vec, target_left_to_right=LEFT_TO_RIGHT_REQUIRED):
    wind_dir = _unit_vector(wind_display_vec)
    if float(np.linalg.norm(wind_dir)) < 1.0e-9:
        raise ValueError("Wind display vector is zero; cannot build view policy.")

    horizontal_norm = float(np.linalg.norm(wind_dir[:2]))
    if horizontal_norm < 1.0e-9:
        raise ValueError(
            "Wind direction is vertical in display coordinates; left-to-right perpendicular policy is undefined."
        )

    wind_azim = math.degrees(math.atan2(float(wind_dir[1]), float(wind_dir[0])))
    azim = wind_azim + 90.0
    elev = CAMERA_ELEV

    forward, right, up = _view_vectors_from_angles(elev, azim)
    wind_projected = wind_dir - float(np.dot(wind_dir, forward)) * forward
    wind_projected_dir = _unit_vector(wind_projected)
    projected_sign = float(np.dot(wind_projected_dir, right))

    if target_left_to_right and projected_sign < 0.0:
        azim += 180.0
        forward, right, up = _view_vectors_from_angles(elev, azim)
        wind_projected = wind_dir - float(np.dot(wind_dir, forward)) * forward
        wind_projected_dir = _unit_vector(wind_projected)
        projected_sign = float(np.dot(wind_projected_dir, right))

    perp_metric = abs(float(np.dot(forward, wind_dir)))
    return {
        "camera_elev_deg": float(elev),
        "camera_azim_deg": float(azim),
        "forward": forward,
        "right": right,
        "up": up,
        "wind_dir": wind_dir,
        "wind_projected_dir": wind_projected_dir,
        "perpendicularity_abs_dot": perp_metric,
        "projected_direction_sign": projected_sign,
    }


def _screen_components(points, center, right, up, forward):
    centered = np.asarray(points, dtype=float) - center
    return centered.dot(right), centered.dot(up), centered.dot(forward)


def _build_wind_arrow_layout(points, view_policy):
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot place wind arrows on empty geometry.")

    center = np.mean(arr, axis=0)
    sx, sy, sd = _screen_components(arr, center, view_policy["right"], view_policy["up"], view_policy["forward"])
    sx_min = float(np.min(sx))
    sx_max = float(np.max(sx))
    span_x = max(sx_max - sx_min, 1.0e-9)
    span_y = max(float(np.max(sy)) - float(np.min(sy)), 1.0e-9)

    required_clearance = max(ARROW_CLEARANCE_RATIO * span_x, ARROW_CLEARANCE_MIN)
    arrow_length = max(0.24 * span_x, 0.80)
    y_values = np.linspace(float(np.min(sy)) + 0.20 * span_y, float(np.max(sy)) - 0.20 * span_y, ARROW_COUNT)
    depth = float(np.median(sd))
    direction_vec = arrow_length * view_policy["wind_projected_dir"]

    factor = 1.0
    best_specs = None
    best_min_clearance = -1.0e9
    while factor <= ARROW_CLEARANCE_MAX_FACTOR + 1.0e-9:
        clearance = factor * required_clearance
        end_x = sx_min - clearance
        start_x = end_x - arrow_length
        specs = []
        min_clearance = float("inf")
        for y_value in y_values:
            start_point = (
                center
                + start_x * view_policy["right"]
                + float(y_value) * view_policy["up"]
                + depth * view_policy["forward"]
            )
            end_point = start_point + direction_vec
            start_sx = float(np.dot(start_point - center, view_policy["right"]))
            end_sx = float(np.dot(end_point - center, view_policy["right"]))
            local_clearance = sx_min - max(start_sx, end_sx)
            min_clearance = min(min_clearance, local_clearance)
            specs.append(
                {
                    "start": start_point,
                    "direction": direction_vec,
                    "start_sx": start_sx,
                    "end_sx": end_sx,
                    "clearance": local_clearance,
                }
            )
        if min_clearance > best_min_clearance:
            best_min_clearance = min_clearance
            best_specs = specs
        if min_clearance >= required_clearance:
            break
        factor *= 1.25

    return best_specs, {
        "required_clearance": required_clearance,
        "min_clearance": best_min_clearance,
        "arrow_count": len(best_specs or []),
    }


def _base_axes_style(ax, view_policy):
    ax.set_axis_off()
    ax.view_init(elev=view_policy["camera_elev_deg"], azim=view_policy["camera_azim_deg"])
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _add_wind_arrows(ax_ref, arrow_specs):
    for spec in arrow_specs:
        direction = spec["direction"]
        start = spec["start"]
        ax_ref.quiver(
            start[0],
            start[1],
            start[2],
            direction[0],
            direction[1],
            direction[2],
            color="dodgerblue",
            linewidth=2.8,
            arrow_length_ratio=0.18,
            alpha=0.95,
        )


def _plot_surface(polygons, scalar_values, cmap_name, title, output_path: Path, view_policy, arrow_specs):
    scalar_array = np.asarray(scalar_values, dtype=float)
    norm = colors.Normalize(vmin=float(np.min(scalar_array)), vmax=float(np.max(scalar_array)))
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    facecolors = cmap(norm(scalar_array))

    fig = plt.figure(figsize=(8.2, 8.6))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(polygons, facecolors=facecolors, linewidths=0.15, edgecolors=(0.1, 0.1, 0.1, 0.18))
    ax.add_collection3d(poly)
    flat_points = [vertex for polygon in polygons for vertex in polygon]

    _add_wind_arrows(ax, arrow_specs)
    _set_view_box(ax, flat_points)
    _base_axes_style(ax, view_policy)
    fig.suptitle(title, fontsize=15, y=0.96)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.66, pad=0.03)
    fig.tight_layout(rect=[0, 0.03, 0.93, 0.95])
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def _case_rows(summary_rows):
    grouped = {}
    for row in summary_rows:
        if row["case_variant"] != "comparison" or row.get("load_case", "combined") != "combined":
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

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 14.5), sharex=True)
    for axis, (key, title, scale, unit) in zip(axes, metrics):
        values = [scale * float(row[key]) for row in refined_rows]
        colors_list = [STATUS_COLORS[row["selection_status"]] for row in refined_rows]
        bars = axis.bar(x, values, color=colors_list)
        for bar, row in zip(bars, refined_rows):
            if row["selection_status"] != STATUS_ENGINEERING:
                bar.set_hatch("//")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=75, ha="right")
    axes[-1].set_xlabel("Scenario/optimizer case")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[key], label=STATUS_LABELS[key])
        for key in [STATUS_ENGINEERING, STATUS_MATH_FALLBACK, STATUS_WARNING]
    ]
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


def _build_case_audit_row(selected_row, wind_axis, view_policy, arrow_metrics):
    perp_pass = float(view_policy["perpendicularity_abs_dot"]) <= VIEW_PERP_DOT_TOL
    direction_pass = float(view_policy["projected_direction_sign"]) > 0.0
    clearance_pass = float(arrow_metrics["min_clearance"]) >= float(arrow_metrics["required_clearance"])
    overall_pass = bool(perp_pass and direction_pass and clearance_pass)

    return {
        "case_id": selected_row["case_id"],
        "case_label": selected_row["case_label"],
        "job_name": selected_row["job_name"],
        "wind_axis": wind_axis,
        "camera_elev_deg": float(view_policy["camera_elev_deg"]),
        "camera_azim_deg": float(view_policy["camera_azim_deg"]),
        "view_dir_x": float(view_policy["forward"][0]),
        "view_dir_y": float(view_policy["forward"][1]),
        "view_dir_z": float(view_policy["forward"][2]),
        "wind_dir_x": float(view_policy["wind_dir"][0]),
        "wind_dir_y": float(view_policy["wind_dir"][1]),
        "wind_dir_z": float(view_policy["wind_dir"][2]),
        "perpendicularity_abs_dot": float(view_policy["perpendicularity_abs_dot"]),
        "perpendicularity_tol": float(VIEW_PERP_DOT_TOL),
        "perpendicular_pass": "yes" if perp_pass else "no",
        "projected_direction_sign": float(view_policy["projected_direction_sign"]),
        "left_to_right_pass": "yes" if direction_pass else "no",
        "min_arrow_clearance": float(arrow_metrics["min_clearance"]),
        "required_arrow_clearance": float(arrow_metrics["required_clearance"]),
        "arrow_clearance_pass": "yes" if clearance_pass else "no",
        "arrow_count": int(arrow_metrics["arrow_count"]),
        "overall_pass": "yes" if overall_pass else "no",
    }


def _plot_case(selected_row, mesh_summary_map, fields_dir: Path, output_dir: Path, wind_axis, wind_display_vec):
    job_name = selected_row["job_name"]
    mesh = _load_json(fields_dir / f"{job_name}_mesh.json")
    static_nodal = _load_csv(fields_dir / f"{job_name}_static_nodal.csv")
    static_element = _load_csv(fields_dir / f"{job_name}_static_element.csv")
    buckle_nodal = _load_csv(fields_dir / f"{job_name}_buckling_mode1_nodal.csv")

    expected = mesh_summary_map[job_name]
    if len(mesh["nodes"]) != int(expected["nodes"]) or len(mesh["elements"]) != int(expected["elements"]):
        raise ValueError(f"Field export count mismatch for {job_name}")

    original_node_map = {
        int(row["node_label"]): {"x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"])} for row in static_nodal
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
        }
        for row in buckle_nodal
    }
    element_stress = {int(row["element_label"]): float(row["mises_pa_avg"]) / 1e6 for row in static_element}

    original_polygons, element_labels = _build_face_polygons(mesh, original_node_map)
    deformed_polygons, _ = _build_face_polygons(mesh, deformed_node_map)
    buckle_polygons, _ = _build_face_polygons(mesh, buckle_node_map)

    stress_values = [element_stress[label] for label in element_labels]
    disp_values = [
        float(np.mean([deformed_node_map[node_id]["umag"] for node_id in element["connectivity"]])) * 1000.0
        for element in mesh["elements"]
    ]
    buckle_values = [
        float(np.mean([buckle_node_map[node_id]["umag"] for node_id in element["connectivity"]]))
        for element in mesh["elements"]
    ]

    flat_points = np.asarray([vertex for polygon in original_polygons for vertex in polygon], dtype=float)
    view_policy = _camera_policy_from_wind(wind_display_vec)
    arrow_specs, arrow_metrics = _build_wind_arrow_layout(flat_points, view_policy)
    audit_row = _build_case_audit_row(selected_row, wind_axis, view_policy, arrow_metrics)
    if audit_row["overall_pass"] != "yes":
        raise ValueError(
            f"Plot view verification failed for {selected_row['case_id']}: "
            f"perp={audit_row['perpendicular_pass']}, ltr={audit_row['left_to_right_pass']}, clearance={audit_row['arrow_clearance_pass']}"
        )

    case_name = selected_row["case_label"]
    _plot_surface(
        original_polygons,
        stress_values,
        "inferno",
        f"{case_name} - Static stress field",
        output_dir / f"task7_{selected_row['case_id']}_stress.png",
        view_policy,
        arrow_specs,
    )
    _plot_surface(
        deformed_polygons,
        disp_values,
        "viridis",
        f"{case_name} - Static displacement field",
        output_dir / f"task7_{selected_row['case_id']}_displacement.png",
        view_policy,
        arrow_specs,
    )
    _plot_surface(
        buckle_polygons,
        buckle_values,
        "cividis",
        f"{case_name} - Buckling mode 1",
        output_dir / f"task7_{selected_row['case_id']}_buckling_mode1.png",
        view_policy,
        arrow_specs,
    )
    return audit_row


def _write_plot_view_audit(audit_csv_path: Path, audit_json_path: Path, rows):
    audit_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "case_label",
                "job_name",
                "wind_axis",
                "camera_elev_deg",
                "camera_azim_deg",
                "view_dir_x",
                "view_dir_y",
                "view_dir_z",
                "wind_dir_x",
                "wind_dir_y",
                "wind_dir_z",
                "perpendicularity_abs_dot",
                "perpendicularity_tol",
                "perpendicular_pass",
                "projected_direction_sign",
                "left_to_right_pass",
                "min_arrow_clearance",
                "required_arrow_clearance",
                "arrow_clearance_pass",
                "arrow_count",
                "overall_pass",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    pass_count = sum(1 for row in rows if row["overall_pass"] == "yes")
    summary = {
        "status": "pass" if pass_count == len(rows) else "fail",
        "cases_total": len(rows),
        "cases_pass": pass_count,
        "perpendicularity_tol": VIEW_PERP_DOT_TOL,
        "left_to_right_required": LEFT_TO_RIGHT_REQUIRED,
        "rows": rows,
    }
    audit_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render Task 7 custom plots from exported Abaqus field data.")
    repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--mesh-summary", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "mesh_summary.csv"))
    parser.add_argument("--summary-csv", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "abaqus_summary.csv"))
    parser.add_argument("--fields-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "fields"))
    parser.add_argument("--config", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "config" / "study_config.json"))
    parser.add_argument("--output-dir", default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "figures"))
    parser.add_argument(
        "--audit-csv",
        default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "plot_view_audit.csv"),
    )
    parser.add_argument(
        "--audit-json",
        default=str(repo_root / "tasks" / "Task_07_Abaqus" / "results" / "data" / "plot_view_audit.json"),
    )
    args = parser.parse_args()

    mesh_summary = _load_csv(Path(args.mesh_summary))
    summary_rows = _load_csv(Path(args.summary_csv))
    config = _load_json(Path(args.config))
    fields_dir = Path(args.fields_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wind_axis = str(config.get("wind", {}).get("wind_direction_axis", "+X"))
    wind_abaqus_vec = _wind_vector_from_axis(wind_axis)
    wind_display_vec = _display_vector(wind_abaqus_vec)

    mesh_summary_map = {row["job_name"]: row for row in mesh_summary}
    row_map = _case_rows(summary_rows)
    refined_rows = [
        _preferred_row(case_id, row_map)
        for case_id in sorted(row_map.keys(), key=lambda value: (int(value.split("_")[0][1:]), value.split("_")[1]))
    ]
    ranked_rows = _rank_rows(refined_rows)
    winner_row = ranked_rows[0]

    _comparison_plot(refined_rows, output_dir / "comparison_metrics.png")
    _warning_plot(refined_rows, output_dir / "s8_warning_metrics.png")
    _winner_mesh_comparison(row_map, winner_row, output_dir / f"task7_{winner_row['case_id']}_mesh_comparison.png")

    audit_rows = []
    audit_csv_path = Path(args.audit_csv)
    audit_json_path = Path(args.audit_json)
    for selected_row in refined_rows:
        audit_row = _plot_case(selected_row, mesh_summary_map, fields_dir, output_dir, wind_axis, wind_display_vec)
        audit_rows.append(audit_row)

    audit_summary = _write_plot_view_audit(audit_csv_path, audit_json_path, audit_rows)
    if audit_summary["status"] != "pass":
        failed_cases = [row["case_id"] for row in audit_rows if row["overall_pass"] != "yes"]
        raise ValueError("Plot view audit failed for cases: " + ", ".join(failed_cases))

    print(
        "Rendered Task 7 custom plots from exported field data with verified view geometry. "
        f"Audit: {audit_csv_path}"
    )


if __name__ == "__main__":
    main()
